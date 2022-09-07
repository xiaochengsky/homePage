let langFlag = 'en'
/**
 *
 * @param {*} className 类名
 * @param {*} content langObj中对应的key的值
 */
const transformLang = (className, content) => {
  const elements = document.getElementsByClassName(className)
  for (let i = 0; i < elements.length; i++) {
    elements[i].innerHTML = langObj[langFlag][content]
  }
}
/**
 * 响应改变语言事件的函数
 */
const onClickChangeLang = () => {
  if (langFlag === 'zh') {
    langFlag = 'en'
  } else {
    langFlag = 'zh'
  }
  allElement.forEach((item) => {
    transformLang(item[0], item[1])
  })
}
